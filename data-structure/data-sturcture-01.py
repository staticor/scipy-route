
# coding: utf-8


import collections

def o(msg=None):
    print( ('*'*30+str(msg)+'*'*30).center(80))


# Stack
# 基本方法: push, pop
# 基本方法: 判断是否为空栈

_stack = []
## push (obj)
_stack.append('a')
assert _stack == ['a']

## pop
_stack.pop(-1)
assert not _stack

## 特点: 后进先出.  (先进后出  FirstIn LastOut)

#带有SIZE限制的栈
#r拥有MAXSIZE属性

# Queue 队列