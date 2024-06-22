from PIL import Image
from transformers import AutoModel, AutoTokenizer
from flask import Flask, request, jsonify, render_template_string
import io
import json
import base64

app = Flask(__name__)

# 모델과 토크나이저를 한 번만 로드
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

# 기본 페이지 라우트 추가
@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>이미지 업로드</title>
    </head>
    <body>
        <h1>이미지를 업로드하여 캡셔닝</h1>
        <form id="uploadForm" enctype="multipart/form-data" action="/upload" method="post">
            <input type="file" id="fileInput" name="file">
            <button type="submit">업로드</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files.length) {
                    alert('파일을 선택하세요!');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (response.ok) {
                        let resultHtml = '<h3>결과:</h3>';
                        resultHtml += '<img src="data:image/png;base64,' + data.image_base64 + '" alt="Uploaded Image" style="max-width: 100%;">';
                        resultHtml += '<p><strong>감정:</strong> ' + data.result.emotions + '</p>';
                        resultHtml += '<p><strong>활동:</strong> ' + data.result.activities + '</p>';
                        resultHtml += '<p><strong>해결책:</strong> ' + data.result.solutions + '</p>';
                        document.getElementById('result').innerHTML = resultHtml;
                    } else {
                        document.getElementById('result').innerText = '오류: ' + data.error;
                    }
                } catch (error) {
                    document.getElementById('result').innerText = '오류: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    ''')

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
        # 오류 메시지를 상세히 출력
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)