#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "broadcast-conversion"

namespace {
class X86BroadcastConversion : public MachineFunctionPass {
public:
  static char ID;
  X86BroadcastConversion() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Broadcast Conversion";
  }

  bool isSlowBroadCast(const X86Subtarget &ST) {
    return ST.hasSlowBroadcast();
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();

    if (!isSlowBroadCast(ST))
      return false;

    bool Changed = false;
    MachineRegisterInfo &MRI = MF.getRegInfo();
    const TargetInstrInfo *TII = ST.getInstrInfo();

    for (MachineBasicBlock &MBB : MF) {
      for (auto MII = MBB.begin(), MIE = MBB.end(); MII != MIE;) {
        MachineInstr &MI = *MII++;

        if (MI.getOpcode() == X86::VBROADCASTSDZ256rm) {
          LLVM_DEBUG(dbgs() << "Checking instr: " << MI);

          int CPIOpIdx = -1;
          for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
            if (MI.getOperand(i).isCPI()) {
              CPIOpIdx = i;
              break;
            }
          }
          if (CPIOpIdx == -1)
            continue;

          int OldCPIdx = MI.getOperand(CPIOpIdx).getIndex();
          MachineConstantPool *MCP = MF.getConstantPool();
          const std::vector<MachineConstantPoolEntry> &Constants =
              MCP->getConstants();
          if (Constants.empty() || OldCPIdx < 0 ||
              OldCPIdx >= (int)Constants.size())
            continue;

          const MachineConstantPoolEntry &CPE = Constants[OldCPIdx];
          if (CPE.isMachineConstantPoolEntry())
            continue;

          // TODO: proceed with other data types.
          const Constant *C = CPE.Val.ConstVal;
          if (!C || !C->getType()->isDoubleTy())
            continue;

          // Transformation example:
          // Before:
          //   .LCPI0_51:                                  <-- Old Index
          //       .quad   0x4010000000000000              <-- 8-byte scalar data
          //   vbroadcastsd .LCPI0_51(%rip), %ymm26        <-- Old Opcode
          //
          // After:
          //   .LCPI0_54:                                  <-- [Step 3] New Index
          //       .quad   0x4010000000000000              <-- [Step 1] Widened 32-byte data
          //       .quad   0x4010000000000000
          //       .quad   0x4010000000000000
          //       .quad   0x4010000000000000
          //   vmovupd .LCPI0_54(%rip), %ymm26             <-- [Step 2] Replaced Opcode

          // [Step 1] Allocate a new 32-byte aligned constant pool entry for the full vector.
          Constant *NewVec = nullptr;
          if (const ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
            double Val = CFP->getValueAPF().convertToDouble();
            NewVec = ConstantDataVector::get(MF.getFunction().getContext(),
                                             SmallVector<double, 4>(4, Val));
          }
          if (!NewVec)
            continue;
          unsigned NewCPIdx = MCP->getConstantPoolIndex(NewVec, Align(32));

          LLVM_DEBUG({
            dbgs() << "\n[Optimization: Broadcast -> Move]\n";
            dbgs() << "Function: " << MF.getName() << "\n";
            dbgs() << "BasicBlock: " << MBB.getName() << "\n";
            dbgs() << "REPLACE: " << MI;
          });

          // [Step 2 & 3] Mutate the instruction in-place: update opcode and constant pool index.
          MI.setDesc(TII->get(X86::VMOVUPDZ256rm));
          MI.getOperand(CPIOpIdx).setIndex(NewCPIdx);

          // Update the MachineMemOperand to describe the new memory reference.
          // We preserve the original PointerInfo and flags, but explicitly expand
          // the memory access size from 8 bytes to 32 bytes with a strict 32-byte
          // alignment guarantee.
          if (!MI.memoperands_empty()) {
            MachineMemOperand *OldMMO = *MI.memoperands_begin();
            MachineMemOperand *NewMMO = MF.getMachineMemOperand(
                OldMMO->getPointerInfo(), OldMMO->getFlags(), 32, Align(32));
            MI.setMemRefs(MF, {NewMMO});
          }

          LLVM_DEBUG(dbgs() << "     TO: " << MI << "\n");
          Changed = true;
        }
      }
    }
    return Changed;
  }
};
} // namespace

char X86BroadcastConversion::ID = 0;
INITIALIZE_PASS(X86BroadcastConversion, "broadcast-conversion",
                "X86 Broadcast Conversion", false, false)

FunctionPass *llvm::createX86BroadcastConversionPass() {
  return new X86BroadcastConversion();
}