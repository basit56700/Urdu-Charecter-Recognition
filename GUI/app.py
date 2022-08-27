from flask import Flask, render_template, Response, request
import cv2
import numpy as np 
import tensorflow as tf

app = Flask(__name__)
video = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/takeimage', methods = ['POST','GET'])
def takeimage():
    name = request.form['name']
   # print(name)
    _, img = video.read()
   
    
    
    model = tf.keras.models.load_model("/tmp/model")
    #model.summary()
    
    
    romanNames=['alif','alif mad aa','bey','pey','tey','ttey',
           'sey','jeem','chay','bari he','khe','daal','dhaal','zaal',
           're','rhey','zaa','zhaa','seen','sheen','svaad','zvaad',
           'toy','zoy','ain','ghain','fey','qaaf','kaaf','gaaf','laam',
           'meem','noon','noonghunna','wao','choti he','do chashmi he',
            'ye','bari ye']
    

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
    resized.shape
    newimg=tf.keras.utils.normalize(resized, axis=1)
    newimg = np.array(newimg).reshape(-1,28,28,1)
    newimg.shape
    predictions = model.predict(newimg)
    pred = np.argmax(predictions)
   # print('class: ', pred)
    print('character: ',romanNames[pred])
 
    pred = romanNames[pred]
    
    cv2.imwrite(f'{pred}.jpg', img)
    
    
    
    return render_template('index.html',pred=pred,status=200)


def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
    app.debug = True
