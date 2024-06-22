from PIL import Image
from transformers import AutoModel, AutoTokenizer
from flask import Flask, request, jsonify, render_template, send_from_directory
import io
import json
import base64
import os

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        question = """What kind of activities, emotions and solutions(about activities and emotions) are in this video?. Answer following JSON format,
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

        # 결과를 JSON으로 파싱
        result_json = json.loads(res)

        return jsonify({'result': result_json, 'image_base64': img_str}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
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
            
            result_json = json.loads(res)
            response_text = f"Emotions: {result_json['emotions']}, Activities: {result_json['activities']}, Solutions: {result_json['solutions']}"
            response_to_response_text = "This is the response to the response."

            return jsonify({'response': response_text, 'response_to_response': response_to_response_text}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

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

            result_json = json.loads(res)
            response_text = f"Emotions: {result_json['emotions']}, Activities: {result_json['activities']}, Solutions: {result_json['solutions']}"
            response_to_response_text = "This is the response to the response."

            return jsonify({'response': response_text, 'response_to_response': response_to_response_text}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
