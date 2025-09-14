from pydantic import ValidationError

def loader_error_msg_formatter(exc: ValidationError, config_path) -> list[dict]:
    error_msg = f"\033[91m\n[INPUT ERROR]\033[0m:\tUnable to read configuration file '{config_path}'.\n"
    for err in exc.errors():
        code = err["type"]

        msg = err["msg"]

        if (len(err["loc"]) > 1 and
            err["loc"][0] == "models_parameter"):
            field = f"\033[91mModel {err['loc'][1]} - {err['loc'][2]}: \033[0m"
        else:
            field = f"\033[91m{err['loc'][0]}: \033[0m"

        error_msg += '\t\t' + field + msg + ' [error: ' + err["type"] + ']\n'

    error_msg += "\t\t\033[91m>>>\033[0m For more details about the validation errors, see: https://docs.pydantic.dev/2.11/errors/validation_errors \n" 
    return error_msg