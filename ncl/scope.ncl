undef("foo")
function foo(i)
local j
begin
    print("i in foo = " + i)
    i = i + 1
    j = i
    return (j)
end

begin
    x = 1
    y = foo(x)
    print("x in main = " + x)
    print("foo(x) returns " + y)
end
