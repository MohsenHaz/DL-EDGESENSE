"""Download all datasets used in the paper (Table 2)."""
import urllib.request, os

URLS = {
    "EnviroSense":   "https://data.mendeley.com/datasets/krkft96n43/2",
    "OpenLORIS":     "https://lifelong-robotic-vision.github.io/dataset/scene.html",
    "ARID":          "https://ieee-dataport.org/open-access/arid-activity-recognition-industrial-domain",
    "NTU-RGB+D":     "https://rose1.ntu.edu.sg/dataset/actionRecognition/",
    "SpectrumSense": "https://www.kaggle.com/datasets/ajithdari/cognitive-radio-spectrum-sensing-dataset",
    "MultiSense":    "https://www.kaggle.com/datasets/maadaaai/mulit-camera-real-scene-video-dataset",
    "EnergyBench":   "https://www.kaggle.com/datasets/elikplim/effficiency-of-energy",
}

def main():
    print("Dataset URLs (manual download required for Kaggle/IEEE Dataport):")
    for name, url in URLS.items():
        print(f"  {name}: {url}")

if __name__ == "__main__":
    main()
