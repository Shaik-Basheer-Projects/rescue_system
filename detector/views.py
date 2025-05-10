from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import RegisterForm, UploadImageForm
from ultralytics import YOLO
import cv2
import os
import pandas as pd

# Load YOLO models
model_v8 = YOLO('C:/Users/Fayaz/basheer/rescue_system/models/yolov8s.pt')
model_v9 = YOLO('C:/Users/Fayaz/basheer/rescue_system/models/yolov9c.pt')

def home(request):
    return redirect('upload')

# Registration view
def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'detector/register.html', {'form': form})

# Login view
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('upload')
        else:
            return render(request, 'detector/login.html', {'error': 'Invalid credentials. Please try again.'})
    return render(request, 'detector/login.html')

# Logout view
def logout_view(request):
    logout(request)
    return redirect('login')

# Only logged-in users can access this view
@login_required
def upload_view(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = request.FILES['image']
            media_dir = 'media'
            os.makedirs(media_dir, exist_ok=True)
            img_path = os.path.join(media_dir, img_file.name)
            with open(img_path, 'wb+') as dest:
                for chunk in img_file.chunks():
                    dest.write(chunk)

            result_img_v8, result_img_v9, df_v8, df_v9 = detect(img_path)

            return render(request, 'detector/result.html', {
                'image_v8': result_img_v8,
                'image_v9': result_img_v9,
                'df_v8': df_v8.to_html(classes='table table-bordered', index=False, border=0),
                'df_v9': df_v9.to_html(classes='table table-bordered', index=False, border=0),
            })
    else:
        form = UploadImageForm()
    return render(request, 'detector/upload.html', {'form': form})

# Detection function (no changes here)
def detect(image_path):
    img = cv2.imread(image_path)

    def run_model(model, color):
        result = model(img)
        img_draw = img.copy()
        detection_data = []
        for r in result:
            for box in r.boxes:
                cls = int(box.cls.cpu().numpy())
                if cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf.item()
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detection_data.append({
                    'Class': model.names[cls],
                    'Confidence': f"{conf:.2f}",
                    'Bounding Box': f"({x1}, {y1}, {x2}, {y2})"
                })
        return img_draw, pd.DataFrame(detection_data)

    img_v8, df_v8 = run_model(model_v8, (255, 0, 0))  # Red for YOLOv8
    img_v9, df_v9 = run_model(model_v9, (0, 255, 0))  # Green for YOLOv9

    out_v8_path = 'media/results_v8.jpg'
    out_v9_path = 'media/results_v9.jpg'
    cv2.imwrite(out_v8_path, img_v8)
    cv2.imwrite(out_v9_path, img_v9)

    return out_v8_path, out_v9_path, df_v8, df_v9
