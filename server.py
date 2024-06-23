from PIL import Image
from transformers import AutoModel, AutoTokenizer
from flask import Flask, request, jsonify, render_template, send_from_directory
import io
import json
import base64
import os
import cv2

app = Flask(__name__)

# 모델과 토크나이저를 한 번만 로드
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

# 정적 파일 경로 설정
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 기본 페이지 라우트 추가
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '비디오 파일이 없습니다'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': '선택된 비디오 파일이 없습니다'}), 400

    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        frames = get_frames_from_video(video_path)

        question = """What kind of activities, emotions and solutions(about activities and emotions) are in this video?. Answer following JSON format,
        {"emotions": emotions, "activities": actions, "solutions": solutions}
        ### Reply JSON but nothing else.###"""
        
        res = analyze_frames(frames, question, model, tokenizer)

        # 결과를 JSON으로 파싱
        print("Model response:", res)  # 응답 디버깅 출력
        try:
            result_json = json.loads(res)
            result_sentence = process_result(result_json)
        except json.JSONDecodeError:
            result_sentence = res  # JSON 파싱 실패 시 원본 응답 사용

        video.seek(0)
        video_base64 = base64.b64encode(video.read()).decode('utf-8')

        return jsonify({'result': result_sentence, 'video_base64': video_base64}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

def get_frames_from_video(video_path, num_frames=3):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [i * (total_frames // num_frames) for i in range(num_frames)]
    
    frames = []
    for frame_index in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image).resize((480,270)))
    
    vidcap.release()
    return frames

def analyze_frames(frames, question, model, tokenizer):
    msgs = [{'role': 'user', 'content': question}]
    res = model.chat(
        msgs=msgs,
        image=frames,
        context=None,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7,
    )
    return res

def process_result(result_json):
    emotions = ', '.join(result_json['emotions']) if isinstance(result_json['emotions'], list) else result_json['emotions']
    activities = ', '.join(result_json['activities']) if isinstance(result_json['activities'], list) else result_json['activities']
    solutions = ', '.join(result_json['solutions']) if isinstance(result_json['solutions'], list) else result_json['solutions']
    result_sentence = f"Emotions: {emotions}\nActivities: {activities}\nSolutions: {solutions}"
    return result_sentence

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    data = request.get_json()
    frames = data['frames']

    try:
        for frame in frames:
            image_data = base64.b64decode(frame.split(',')[1])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            question = """What kind of activities, emotions and solutions(about activities and emotions) are in this image?. Answer following JSON format,
            {"emotions": emotions, "activities": actions, "solutions": solutions}
            ### Reply JSON but nothing else.###"""
            msgs = [{'role': 'user', 'content': question}]
            
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7,
            )

            print("Model response:", res)  # 응답 디버깅 출력
            try:
                result_json = json.loads(res)
                result_sentence = process_result(result_json)
            except json.JSONDecodeError:
                result_sentence = res  # JSON 파싱 실패 시 원본 응답 사용

            response_to_response_text = "This is the response to the response."

            return jsonify({'response': result_sentence, 'response_to_response': response_to_response_text, 'image_base64': frame}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
