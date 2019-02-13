from torrequest import TorRequest

with TorRequest() as tr:
    response = tr.get('http://ipecho.net/plain')
    print(response.text)