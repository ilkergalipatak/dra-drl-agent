import requests

url = "https://ollama.parts-soft.net/api/tags"  # endpointi değiştir

headers = {
    "Authorization": "Bearer 9bb67e0f507743270aa7289fbb970bc088761b448e78368c28ebda15d7f7d4ec",
    "Content-Type": "application/json",
}

# GET isteği örneği
response = requests.get(url, headers=headers)

print(response.status_code)
print(response.text)
