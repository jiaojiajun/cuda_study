1. baseline
2. without_mode 在原有的基础上去除线程索引的mod操作
3. without_bank_conflict 在原有的基础上减少访问共享内存的冲突
4. half_blocks.cu 在原有的基础上减少block的数量
5. device_function.cu 在上一步的基础上减少循环次数，当规约至一个线程束（warp，32线程）的时候不再循环，而是手动写出循环执行过程。
