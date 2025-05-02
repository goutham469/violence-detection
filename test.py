from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/" , methods=['GET'])
def root() :
    return "Hi test server running"

app.run( host='0.0.0.0' , port=80 , debug=True )