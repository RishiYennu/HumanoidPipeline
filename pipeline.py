import sys
import time
from google import genai
from google.genai import types
from PIL import Image
import subprocess
import os


#Create a file within the folder gemini_files and name is apiKey and paste only your Gemini API key in there.
f = open("gemini_files/apiKey", "r")
client = genai.Client(api_key=f.read())

def run(cmd, cwd=None, env=None):
    """Run a shell command and stream output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"WARNING: Command exited with code {result.returncode}")
    return result.returncode


def main():
  args = sys.argv[1:]

  if len(args) > 1:
    print("The command for running the script is: python pipeline.py [simulation image]")
    return
  
  image = sys.argv[1]

  human_image = generate_human_image(image)

  generate_video(human_image)


def generate_human_image(robot_image : str):
  # Edit Image prompt in the below file
  image_prompt = open("gemini_files/imagePrompt.txt", "r")
  img = Image.open(robot_image)

  response = client.models.generate_content(
      model="gemini-3-pro-image-preview",
      contents=[image_prompt.read(), img],
      config=types.GenerateContentConfig(
          response_modalities=["TEXT", "IMAGE"],
      ),
  )

  for part in response.candidates[0].content.parts:
      if part.inline_data:
          with open("output/human_video_image.png", "wb") as f:
              f.write(part.inline_data.data)
          print("Saved human_video_image.png")
      elif part.text:
          print(part.text)
  
  return "human_video_image.png"

def generate_video(human_image : str): 
  # Edit video prompt in the below file
  image_file = client.files.upload(file="human_video_image.png")

  operation = client.models.generate_videos(
      model="veo-2.0-generate-001",
      image=image_file,
      config=types.GenerateVideosConfig(
          prompt="Describe the motion/animation you want",  
          number_of_videos=1,
          duration_seconds=5, 
          negative_prompt="blurry, low quality",  
      ),
  )
  while not operation.done:
      time.sleep(10)
      operation = client.operations.get(operation)

  for video in operation.result.generated_videos:
      client.files.download(file=video.video, download_path=f"output/output.mp4")
      print(f"Saved output.mp4")


if __name__ == "__main__":
  main()