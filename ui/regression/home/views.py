from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image
from io import BytesIO
from . import helper
import base64

matplotlib.use('Agg')
dataSet = helper.load_from_api()
context = {
    "startDate": 0,
    "duration": 0,
}

def addVals(request):
    context["startDate"] = int(request.POST.get('start_date'))
    context["duration"] = int(request.POST.get('duration'))
    print("Start Date: ", context["startDate"], "  ", context["duration"])
    predicted, actual, accuracy, top10pred, top10actual = helper.master_run_func(context["startDate"], context["duration"], dataSet)
    context["accuracy"] = sum(accuracy)/len(accuracy)
    context["pred"] = top10pred
    context["actual"] = top10actual
    return render(request, 'home/image.html', context)

def showimage(request):
    predicted, actual, accuracy, top10pred, top10actual = helper.master_run_func(context["startDate"], context["duration"], dataSet)
    xvalue = actual
    yvalue = predicted
    plt.style.use('seaborn')
    plt.scatter(xvalue, yvalue, alpha = 0.75)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()
 
    # Store image in a string buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
 
    # Send buffer in a http response the the browser with the mime type image/png set
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def index(request):
    if request.POST.get('generate_image'):
        return addVals(request)
    return render(request, 'home/index.html')
