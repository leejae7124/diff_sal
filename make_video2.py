from moviepy.editor import VideoFileClip, AudioFileClip

# 파일 경로 설정
video_path = './heatmap/video/S01E01_000_heatmap_video.mp4'
audio_path = './data/video_audio_MEmoR/S01E01_000.wav'
output_path = './heatmap/video/S01E01_000_heatmap_with_audio.mp4'

# 비디오와 오디오 불러오기
video_clip = VideoFileClip(video_path)
audio_clip = AudioFileClip(audio_path)

# 비디오에 오디오 추가
final_clip = video_clip.set_audio(audio_clip)

# 최종 비디오 저장
final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
