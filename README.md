# JobLens Backend

This is the backend service for JobLens, a tool that analyzes resumes and job descriptions to provide match scores, skill recommendations, and more.

## Features

- Resume and job description analysis
- Skill extraction and matching
- Match score calculation
- Tailored resume generation
- Cover letter generation
- Improvement suggestions

## Deployment on Render

### Prerequisites

- A Render account
- OpenAI API key for AI-powered features

### Steps to Deploy

1. **Create a new Web Service on Render**
   - Connect your GitHub repository
   - Select the repository containing this code

2. **Configure the Web Service**
   - **Name**: Choose a name for your service (e.g., joblens-backend)
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: Use the command from the Procfile: `gunicorn --bind 0.0.0.0:$PORT --workers=2 --threads=4 --timeout=120 app:app`

3. **Add Environment Variables**
   - `OPENAI_API_KEY`: Your OpenAI API key

4. **Advanced Settings**
   - Set the instance type to at least 1GB RAM (Basic plan or higher)
   - Increase the timeout to 120 seconds

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your application

## Performance Optimizations

The application has been optimized for deployment on Render with the following improvements:

1. **Lazy Loading of NLP Models**: Models are only loaded when needed
2. **Limited Text Processing**: Text length is capped to prevent memory issues
3. **Optimized Worker Configuration**: Using 2 workers with 4 threads each
4. **File Size Limits**: Preventing large file uploads
5. **Error Handling**: Comprehensive error handling to prevent crashes
6. **Response Size Limits**: Limiting the number of skills returned

## Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables: `export OPENAI_API_KEY=your_api_key`
3. Run the application: `python app.py`

## Troubleshooting Render Deployment

- If you encounter memory issues, try reducing the number of workers in the Procfile
- For timeout errors, increase the timeout value in the Procfile
- Check Render logs for specific error messages