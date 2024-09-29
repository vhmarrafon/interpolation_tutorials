import requests
import time

def retry_api_call(max_tries=5):
    '''
    Decorator to retry an API call if it fails.

    :param max_tries: int - number of maximum tries before raising an error
    '''

    def decorator(func):
        def wrapper(*args, current_try=0, **kwargs):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                if current_try >= max_tries:
                    raise RuntimeError(f"Max tries attempted in API call, please try again") from e
                else:
                    time.sleep(1)
                    return wrapper(*args, current_try=current_try + 1, **kwargs)

        return wrapper

    return decorator
