WHAT I DID
* Ran BI (ShortGPT's layer-importance score) on 3 small models: Qwen3-0.6B, Qwen2.5-0.5B and BLOOM-560M.
* Used the same 997 FLORES+ sentences in 6 languages: English, Hindi, Bengali, Marathi, Tamil and Telugu.
* For each language, found which layers look least important, i.e. the ones ShortGPT would remove.
* Used Shridhar's bi.py as-is, with a small script (run_bi.py) around it to load data, run everything and save results.

ONE TECHNICAL FIX
In half precision (fp16), the model's small rounding errors were big enough to slightly change the results. In full precision (fp32) they disappear. So I now run everything in fp32. It still fits easily on free Colab.

WHAT WE FOUND
* Qwen3: English and the Indian languages would remove different layers. English would remove mostly late layers (23–26); all five Indian languages would remove middle layers instead. Only 3 of the 7 layers are the same. The late layers English would remove are 2–4× more active for Indian languages.
* Qwen2.5: same trend but smaller. English and the Indian languages agree more.
* BLOOM: each Indian language behaves differently. Hindi is closest to English, Telugu is furthest.
* Qwen splits Indian-language text into 4–7× more pieces than English, while BLOOM splits it almost like English. BLOOM still shows differences, so this isn't only caused by how the text is split.
* Using just the first 64 sentences gave the same layers to remove as using all 997, so the result is stable.

https://github.com/KunalAyush1/inference-script/blob/main/InferenceScript.ipynbNOTE
So far we have only measured which layers look important. We haven't removed any layers yet, so we don't yet know if Indian languages actually get worse. That's the next step
