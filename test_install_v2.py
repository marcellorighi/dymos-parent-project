import openmdao.api as om

p = om.Problem()
p.driver = om.pyOptSparseDriver()

# This should NOT raise an error:
p.driver.options['optimizer'] = 'IPOPT'

# If we get here, it worked!
print("✓ IPOPT is available!")

# Show available IPOPT options
print("\\nAvailable IPOPT settings:")
for key in sorted(p.driver.opt_settings.keys()):
    print(f"  - {key}")
