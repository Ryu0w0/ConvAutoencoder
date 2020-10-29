def notify(message):
    import requests
    url = "https://notify-api.line.me/api/notify"
    token = "KK30I72oh8Dc4fxPQVkldnJb6sBeaTaOWIMmVAoSObF"
    headers = {"Authorization": "Bearer " + token}

    message = message
    payload = {"message": message}

    requests.post(url, headers=headers, params=payload)


is_from_onedrive = False
def root_changer(path):
    if is_from_onedrive:
        return path.replace("./", "C:/Users/adbb261/OneDrive - City, University of London/")
    return path
