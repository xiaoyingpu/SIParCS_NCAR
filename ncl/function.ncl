undef("function_name")  ; optional, can't undefine built-in functions

function func_name(x1, x2)
local sx1, var
begin
    sx1 = x1 * x2
    var = sx1 * x2
    var@long_name = "result"
    var@units = "degree_C"
    return (var)
end


;; somewhere else
;load "./some/path/myLib.ncl"
begin
    x = func_name(1,2)
    print(x)
end
