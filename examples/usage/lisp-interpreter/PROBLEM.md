# Build a tiny Lisp interpreter in TypeScript

Write a file called `lisp.ts` that implements a small Lisp interpreter.

## Requirements

1. **Tokenizer** — split input into tokens (parens, numbers, symbols, strings).
2. **Parser** — convert tokens into a nested AST (arrays for lists, primitives for atoms).
3. **Evaluator** — walk the AST with an environment supporting:
   - Arithmetic: `+`, `-`, `*`, `/`
   - Comparison: `<`, `>`, `=`
   - `define` — bind a value in the current environment
   - `if` — conditional
   - `lambda` — closures
   - `list`, `car`, `cdr`, `cons`
   - `print` — output to stdout
4. **REPL** — if run with no args, start a read-eval-print loop on stdin.
   If a filename arg is given, execute that file instead.

## Testing

Create a test file `test.lisp` that exercises every feature:

```lisp
(define square (lambda (x) (* x x)))
(print (square 7))

(define fact (lambda (n)
  (if (= n 0) 1 (* n (fact (- n 1))))))
(print (fact 10))

(define xs (list 1 2 3))
(print (car xs))
(print (cdr xs))
(print (cons 0 xs))

(print (if (> 3 2) "yes" "no"))
```

Expected output:
```
49
3628800
1
(2 3)
(0 1 2 3)
yes
```

Run tests with: `bun run lisp.ts test.lisp`
