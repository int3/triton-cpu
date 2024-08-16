from .cuda import libdevice as cuda_libdevice
from .hip import libdevice as hip_libdevice
from triton.language import core
from typing import TypeVar

T = TypeVar('T')


def dispatch(fn: T) -> T:
    """Dispatch a function to a correct implementation."""
    assert callable(fn)

    @core.builtin
    def resolver(*args, _builder, **kwargs):
        _backend = _builder.options.backend_name
        if _backend == 'cuda':
            _curr_libdevice_module = cuda_libdevice
        elif _backend == 'hip':
            _curr_libdevice_module = hip_libdevice
        else:
            raise RuntimeError('unknown backend')

        try:
            _impl = getattr(_curr_libdevice_module, fn.__name__)
        except AttributeError:
            raise RuntimeError(f'`{_backend}` does not provide support for `{fn.__name__}` extra function')

        return _impl

    return core.resolved_by(resolver)(fn)


@dispatch
def clz(arg0):
    ...


@dispatch
def popc(arg0):
    ...


@dispatch
def byte_perm(arg0, arg1, arg2):
    ...


@dispatch
def mulhi(arg0, arg1):
    ...


@dispatch
def mul24(arg0, arg1):
    ...


@dispatch
def brev(arg0):
    ...


@dispatch
def sad(arg0, arg1, arg2):
    ...


@dispatch
def abs(arg0):
    ...


@dispatch
def floor(arg0):
    ...


@dispatch
def rcp64h(arg0):
    ...


@dispatch
def rsqrt(arg0):
    ...


@dispatch
def ceil(arg0):
    ...


@dispatch
def trunc(arg0):
    ...


@dispatch
def exp2(arg0):
    ...


@dispatch
def saturatef(arg0):
    ...


@dispatch
def fma_rn(arg0, arg1, arg2):
    ...


@dispatch
def fma_rz(arg0, arg1, arg2):
    ...


@dispatch
def fma_rd(arg0, arg1, arg2):
    ...


@dispatch
def fma_ru(arg0, arg1, arg2):
    ...


@dispatch
def fast_dividef(arg0, arg1):
    ...


@dispatch
def div_rn(arg0, arg1):
    ...


@dispatch
def div_rz(arg0, arg1):
    ...


@dispatch
def div_rd(arg0, arg1):
    ...


@dispatch
def div_ru(arg0, arg1):
    ...


@dispatch
def rcp_rn(arg0):
    ...


@dispatch
def rcp_rz(arg0):
    ...


@dispatch
def rcp_rd(arg0):
    ...


@dispatch
def rcp_ru(arg0):
    ...


@dispatch
def sqrt_rn(arg0):
    ...


@dispatch
def sqrt_rz(arg0):
    ...


@dispatch
def sqrt_rd(arg0):
    ...


@dispatch
def sqrt_ru(arg0):
    ...


@dispatch
def sqrt(arg0):
    ...


@dispatch
def add_rn(arg0, arg1):
    ...


@dispatch
def add_rz(arg0, arg1):
    ...


@dispatch
def add_rd(arg0, arg1):
    ...


@dispatch
def add_ru(arg0, arg1):
    ...


@dispatch
def mul_rn(arg0, arg1):
    ...


@dispatch
def mul_rz(arg0, arg1):
    ...


@dispatch
def mul_rd(arg0, arg1):
    ...


@dispatch
def mul_ru(arg0, arg1):
    ...


@dispatch
def double2float_rn(arg0):
    ...


@dispatch
def double2float_rz(arg0):
    ...


@dispatch
def double2float_rd(arg0):
    ...


@dispatch
def double2float_ru(arg0):
    ...


@dispatch
def double2int_rn(arg0):
    ...


@dispatch
def double2int_rz(arg0):
    ...


@dispatch
def double2int_rd(arg0):
    ...


@dispatch
def double2int_ru(arg0):
    ...


@dispatch
def double2uint_rn(arg0):
    ...


@dispatch
def double2uint_rz(arg0):
    ...


@dispatch
def double2uint_rd(arg0):
    ...


@dispatch
def double2uint_ru(arg0):
    ...


@dispatch
def int2double_rn(arg0):
    ...


@dispatch
def uint2double_rn(arg0):
    ...


@dispatch
def float2int_rn(arg0):
    ...


@dispatch
def float2int_rz(arg0):
    ...


@dispatch
def float2int_rd(arg0):
    ...


@dispatch
def float2int_ru(arg0):
    ...


@dispatch
def float2uint_rn(arg0):
    ...


@dispatch
def float2uint_rz(arg0):
    ...


@dispatch
def float2uint_rd(arg0):
    ...


@dispatch
def float2uint_ru(arg0):
    ...


@dispatch
def int2float_rn(arg0):
    ...


@dispatch
def int2float_rz(arg0):
    ...


@dispatch
def int2float_rd(arg0):
    ...


@dispatch
def int2float_ru(arg0):
    ...


@dispatch
def uint2float_rn(arg0):
    ...


@dispatch
def uint2float_rz(arg0):
    ...


@dispatch
def uint2float_rd(arg0):
    ...


@dispatch
def uint2float_ru(arg0):
    ...


@dispatch
def hiloint2double(arg0, arg1):
    ...


@dispatch
def double2loint(arg0):
    ...


@dispatch
def double2hiint(arg0):
    ...


@dispatch
def float2ll_rn(arg0):
    ...


@dispatch
def float2ll_rz(arg0):
    ...


@dispatch
def float2ll_rd(arg0):
    ...


@dispatch
def float2ll_ru(arg0):
    ...


@dispatch
def float2ull_rn(arg0):
    ...


@dispatch
def float2ull_rz(arg0):
    ...


@dispatch
def float2ull_rd(arg0):
    ...


@dispatch
def float2ull_ru(arg0):
    ...


@dispatch
def double2ll_rn(arg0):
    ...


@dispatch
def double2ll_rz(arg0):
    ...


@dispatch
def double2ll_rd(arg0):
    ...


@dispatch
def double2ll_ru(arg0):
    ...


@dispatch
def double2ull_rn(arg0):
    ...


@dispatch
def double2ull_rz(arg0):
    ...


@dispatch
def double2ull_rd(arg0):
    ...


@dispatch
def double2ull_ru(arg0):
    ...


@dispatch
def ll2float_rn(arg0):
    ...


@dispatch
def ll2float_rz(arg0):
    ...


@dispatch
def ll2float_rd(arg0):
    ...


@dispatch
def ll2float_ru(arg0):
    ...


@dispatch
def ull2float_rn(arg0):
    ...


@dispatch
def ull2float_rz(arg0):
    ...


@dispatch
def ull2float_rd(arg0):
    ...


@dispatch
def ull2float_ru(arg0):
    ...


@dispatch
def ll2double_rn(arg0):
    ...


@dispatch
def ll2double_rz(arg0):
    ...


@dispatch
def ll2double_rd(arg0):
    ...


@dispatch
def ll2double_ru(arg0):
    ...


@dispatch
def ull2double_rn(arg0):
    ...


@dispatch
def ull2double_rz(arg0):
    ...


@dispatch
def ull2double_rd(arg0):
    ...


@dispatch
def ull2double_ru(arg0):
    ...


@dispatch
def int_as_float(arg0):
    ...


@dispatch
def float_as_int(arg0):
    ...


@dispatch
def uint_as_float(arg0):
    ...


@dispatch
def float_as_uint(arg0):
    ...


@dispatch
def longlong_as_double(arg0):
    ...


@dispatch
def double_as_longlong(arg0):
    ...


@dispatch
def fast_sinf(arg0):
    ...


@dispatch
def fast_cosf(arg0):
    ...


@dispatch
def fast_log2f(arg0):
    ...


@dispatch
def fast_logf(arg0):
    ...


@dispatch
def fast_expf(arg0):
    ...


@dispatch
def fast_tanf(arg0):
    ...


@dispatch
def fast_exp10f(arg0):
    ...


@dispatch
def fast_log10f(arg0):
    ...


@dispatch
def fast_powf(arg0, arg1):
    ...


@dispatch
def hadd(arg0, arg1):
    ...


@dispatch
def rhadd(arg0, arg1):
    ...


@dispatch
def sub_rn(arg0, arg1):
    ...


@dispatch
def sub_rz(arg0, arg1):
    ...


@dispatch
def sub_rd(arg0, arg1):
    ...


@dispatch
def sub_ru(arg0, arg1):
    ...


@dispatch
def rsqrt_rn(arg0):
    ...


@dispatch
def ffs(arg0):
    ...


@dispatch
def rint(arg0):
    ...


@dispatch
def llrint(arg0):
    ...


@dispatch
def nearbyint(arg0):
    ...


@dispatch
def isnan(arg0):
    ...


@dispatch
def signbit(arg0):
    ...


@dispatch
def copysign(arg0, arg1):
    ...


@dispatch
def finitef(arg0):
    ...


@dispatch
def isinf(arg0):
    ...


@dispatch
def nextafter(arg0, arg1):
    ...


@dispatch
def sin(arg0):
    ...


@dispatch
def cos(arg0):
    ...


@dispatch
def sinpi(arg0):
    ...


@dispatch
def cospi(arg0):
    ...


@dispatch
def tan(arg0):
    ...


@dispatch
def log2(arg0):
    ...


@dispatch
def exp(arg0):
    ...


@dispatch
def exp10(arg0):
    ...


@dispatch
def cosh(arg0):
    ...


@dispatch
def sinh(arg0):
    ...


@dispatch
def tanh(arg0):
    ...


@dispatch
def atan2(arg0, arg1):
    ...


@dispatch
def atan(arg0):
    ...


@dispatch
def asin(arg0):
    ...


@dispatch
def acos(arg0):
    ...


@dispatch
def log(arg0):
    ...


@dispatch
def log10(arg0):
    ...


@dispatch
def log1p(arg0):
    ...


@dispatch
def acosh(arg0):
    ...


@dispatch
def asinh(arg0):
    ...


@dispatch
def atanh(arg0):
    ...


@dispatch
def expm1(arg0):
    ...


@dispatch
def hypot(arg0, arg1):
    ...


@dispatch
def rhypot(arg0, arg1):
    ...


@dispatch
def norm3d(arg0, arg1, arg2):
    ...


@dispatch
def rnorm3d(arg0, arg1, arg2):
    ...


@dispatch
def norm4d(arg0, arg1, arg2, arg3):
    ...


@dispatch
def rnorm4d(arg0, arg1, arg2, arg3):
    ...


@dispatch
def cbrt(arg0):
    ...


@dispatch
def rcbrt(arg0):
    ...


@dispatch
def j0(arg0):
    ...


@dispatch
def j1(arg0):
    ...


@dispatch
def y0(arg0):
    ...


@dispatch
def y1(arg0):
    ...


@dispatch
def yn(arg0, arg1):
    ...


@dispatch
def jn(arg0, arg1):
    ...


@dispatch
def cyl_bessel_i0(arg0):
    ...


@dispatch
def cyl_bessel_i1(arg0):
    ...


@dispatch
def erf(arg0):
    ...


@dispatch
def erfinv(arg0):
    ...


@dispatch
def erfc(arg0):
    ...


@dispatch
def erfcx(arg0):
    ...


@dispatch
def erfcinv(arg0):
    ...


@dispatch
def normcdfinv(arg0):
    ...


@dispatch
def normcdf(arg0):
    ...


@dispatch
def lgamma(arg0):
    ...


@dispatch
def ldexp(arg0, arg1):
    ...


@dispatch
def scalbn(arg0, arg1):
    ...


@dispatch
def fmod(arg0, arg1):
    ...


@dispatch
def remainder(arg0, arg1):
    ...


@dispatch
def fma(arg0, arg1, arg2):
    ...


@dispatch
def pow(arg0, arg1):
    ...


@dispatch
def tgamma(arg0):
    ...


@dispatch
def round(arg0):
    ...


@dispatch
def llround(arg0):
    ...


@dispatch
def fdim(arg0, arg1):
    ...


@dispatch
def ilogb(arg0):
    ...


@dispatch
def logb(arg0):
    ...


@dispatch
def isfinited(arg0):
    ...
