; SPDX-License-Identifier: GPL-2.0-only
; Copyright (C) 2026 Adam55Lord
; Lord Core sample library: ltmath.llf  (nasm -f bin)
; exports: add(a, b) @ 0x00, square(x) @ 0x10; LibraryMain(reason) @ 0x20
; LT ABI: syscall 2 = write(ptr, len) to the kernel log.
BITS 64
add:
    lea rax, [rdi + rsi]
    ret
times 0x10 - ($ - $$) db 0xCC
square:
    mov rax, rdi
    imul rax, rdi
    ret
times 0x20 - ($ - $$) db 0xCC
libmain:
    cmp rdi, 1
    je .load
    mov eax, 2
    lea rdi, [rel msg_unload]
    mov esi, msg_unload_len
    syscall
    ret
.load:
    mov eax, 2
    lea rdi, [rel msg_load]
    mov esi, msg_load_len
    syscall
    ret
msg_load:   db "[ltmath] LibraryMain: loaded"
msg_load_len: equ $ - msg_load
msg_unload: db "[ltmath] LibraryMain: unloaded"
msg_unload_len: equ $ - msg_unload
