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
