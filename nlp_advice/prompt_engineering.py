from transformers import pipeline

# Lightweight baseline; will download on first run
generator = pipeline('text-generation', model='gpt2')

def advise(features):
    heart_rate, sleep_hours, activity_level, temperature = features
    prompt = (
        f'User heart rate: {heart_rate} bpm, sleep: {sleep_hours} hours, '
        f'activity level: {activity_level}/10, temperature: {temperature} °C. '
        'Give concise, safe, actionable, non-medical lifestyle advice.'
    )
    out = generator(prompt, max_length=120, num_return_sequences=1)
    return out[0]['generated_text']

if __name__ == '__main__':
    print(advise([120, 4, 2, 37.9]))
