import scipy
from transformers import AutoProcessor, SeamlessM4Tv2Model


def main():
    print("Hello from translation-api!")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    # from text
    text_inputs = processor(
        text="Hello, my dog is cute", src_lang="eng", return_tensors="pt"
    )
    audio_array_from_text = (
        model.generate(**text_inputs, tgt_lang="jpn")[0].cpu().numpy().squeeze()
    )

    sample_rate = model.config.sampling_rate

    scipy.io.wavfile.write(
        "output/output_from_text.wav", rate=sample_rate, data=audio_array_from_text
    )


if __name__ == "__main__":
    main()
