import json
from datetime import datetime
from flask import Flask, jsonify, make_response, request, render_template
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs
from fpdf import FPDF

app = Flask(__name__)

@app.route('/')
def summary_page():
    return render_template('popup.html')

@app.route('/time', methods=['GET'])
def get_time():
    return str(datetime.now())

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get('v')
    if video_id:
        return video_id[0]
    else:
        return None

@app.route('/summarize/check', methods=['GET'])
def transcript():
    youtube_url = request.args.get('youtube_url')
    if not youtube_url:
        return jsonify({'error': 'YouTube URL is missing'}), 400

    # Parse the YouTube URL to extract the video ID
    parsed_url = urlparse(youtube_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get('v') or [None]
    if not video_id[0]:
        # Attempt to extract video ID from shortened URL format
        path_segments = parsed_url.path.split('/')
        video_id = path_segments[-1] if path_segments[-1] else path_segments[-2]

    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL. Please provide a valid YouTube video URL.'}), 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        
        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=text)
        pdf_output = pdf.output(dest='S').encode('latin1')
        
        # Prepare download response
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=transcript.pdf'
        return response, 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize/summary', methods=['GET'])
def summarize_summary():
    youtube_url = request.args.get('youtube_url')
    if not youtube_url:
        return jsonify({'error': 'YouTube URL is missing'}), 400

    # Parse the YouTube URL to extract the video ID
    parsed_url = urlparse(youtube_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get('v') or [None]
    if not video_id[0]:
        # Attempt to extract video ID from shortened URL format
        path_segments = parsed_url.path.split('/')
        video_id = path_segments[-1] if path_segments[-1] else path_segments[-2]

    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])

        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base')

        inputs = tokenizer.encode("summarize:" + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                  no_repeat_ngram_size=2, num_return_sequences=4,
                                  early_stopping=True)
        summaries = [tokenizer.decode(output).replace("</s>", "").strip() for output in outputs]

        unique_summaries = list(set(summaries))

        return jsonify(unique_summaries), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
