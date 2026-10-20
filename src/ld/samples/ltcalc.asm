; SPDX-License-Identifier: GPL-2.0-only
; Copyright (C) 2026 Adam55Lord
; Lord Core sample program: ltcalc.lxf  (nasm -f bin)
; Uses a relocation (msg_ptr @ 0x100) and two imports from ltmath.llf
; (IAT slots @ 0x108 and 0x110). Returns square(add(10, 32)) = 1764, which becomes the
; exit code; returns 1 if the relocated pointer did not work.
BITS 64
_start:
    mov eax, 2
    mov rdi, [rel msg_ptr]
    mov esi, msg_len
    syscall
    cmp rax, msg_len
    jne .bad
    mov edi, 10
    mov esi, 32
    call [rel iat_add]
    mov rdi, rax
    call [rel iat_square]
    ret
.bad:
    mov eax, 1
    ret
times 0x100 - ($ - $$) db 0xCC
msg_ptr:    dq msg
iat_add:    dq 0
iat_square: dq 0
msg: db "[ltcalc] hello from an LXF program"
msg_len: equ $ - msg
