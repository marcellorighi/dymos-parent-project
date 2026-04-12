# Check if pyoptsparse is installed at all
try:
    import pyoptsparse
    print(f"✓ pyoptsparse is installed")
    print(f"  Version: {pyoptsparse.__version__}")
    print(f"  Location: {pyoptsparse.__file__}")
except ImportError:
    print("✗ pyoptsparse is NOT installed")
    exit()

# Check which optimizers are available
print("\\nChecking available optimizers:")

optimizers_to_check = ['SLSQP', 'IPOPT', 'SNOPT', 'ALPSO', 'NSGA2']

for opt_name in optimizers_to_check:
    try:
        optimizer_class = getattr(pyoptsparse, opt_name)
        print(f"  ✓ {opt_name} is available")
    except AttributeError:
        print(f"  ✗ {opt_name} is NOT available")

# More detailed IPOPT check
print("\\nDetailed IPOPT check:")
try:
    from pyoptsparse import IPOPT
    print("  ✓ IPOPT can be imported")
    print(f"  IPOPT object: {IPOPT}")
except ImportError as e:
    print(f"  ✗ IPOPT import failed: {e}")
except Exception as e:
    print(f"  ✗ Unexpected error: {e}")
