# Streamlit Workshop

## Setup

### Docker-way

Just run `docker-compose up app`

### Standard way

Install dependencies `python3 -m pip install -r requirements.app.txt `.

Start the app `python3 -m streamlit run app/main.py`.

In both cases, you should see something like


```bash
You can now view your Streamlit app in your browser.

URL: http://0.0.0.0:80
```

At first run, it can ask you for your email, just skip it with Enter.

Open http://0.0.0.0:80 in your browser and you should see a pretty welcome message.
