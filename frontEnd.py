import os
#import magic
import urllib.request
#from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from backEnd import main

from flask import Flask

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('index.html', variable=processFile())

def processFile():
    try:
        fishName=main()
        print(fishName)
        return fishName
        os.remove("fish.png")
    except:
        return ""

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			temp=filename.split(".")
			temp[0]="fish"
			filename=temp[0]+"."+temp[1]
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File successfully uploaded')
			flash(processFile())
			return redirect('/')
		else:
			flash('Allowed file types are png, jpg, jpeg')
			return redirect(request.url)

def show_index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'fish.png')
    return render_template("index.html", user_image = full_filename)

if __name__ == "__main__":
    app.run(debug=True)
