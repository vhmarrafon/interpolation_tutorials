

def check_parameters(nc_engine, domain, res, start_date, end_date, variables):
    if nc_engine not in ['netcdf4', 'xarray']:
        raise ValueError(f"invalid NC_ENGINE @ {nc_engine} choose 'xarray' or 'netcdf4'")

    # Check DOMAIN
    if domain[0] >= domain[1]:
        raise ValueError(f"invalid domain, initial longitude {domain[0]} must be lower than final one {domain[1]}")

    if domain[2] >= domain[3]:
        raise ValueError(f"invalid domain, initial latitude {domain[2]} must be lower than final one {domain[3]}")

    if res <= 0:
        raise ValueError(f"invalid resolution, resolution must be higher than 0, {res} passed")

    # Check date
    if start_date > end_date:
        raise ValueError(f"invalid date definition, start date {start_date} must be higher or equal end date {end_date}")

    # Check if variables are defined
    if not variables:
        raise ValueError(f"VARIABLES must be defined")

    return True