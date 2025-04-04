# Deploying Your Data Visualization App to Vercel

This guide will walk you through the process of deploying your application to Vercel.

## Prerequisites

1. **Vercel Account**: [Sign up for free](https://vercel.com/signup)
2. **MongoDB Atlas Account**: [Sign up for free](https://www.mongodb.com/cloud/atlas/register)
3. **Pinecone Account**: For vector storage ([Sign up](https://www.pinecone.io/))
4. **OpenAI API Key**: For AI functionality

## Step 1: Set Up MongoDB Atlas

1. Create a new cluster (free tier is fine)
2. Create a database user with read/write permissions
3. Add your IP address to the IP whitelist (or use 0.0.0.0/0 for anywhere)
4. Get your MongoDB connection string which looks like:
   ```
   mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority
   ```

## Step 2: Deploy to Vercel

### Option A: Using Vercel CLI

1. Install the Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Log in to your Vercel account:
   ```
   vercel login
   ```

3. From your project directory, run:
   ```
   vercel
   ```

4. When prompted, set up your project with the following options:
   - Set the build output directory to `./`
   - Add the environment variables from `.env.vercel`

### Option B: Using Vercel Web Interface

1. Push your code to GitHub
2. Go to [Vercel Dashboard](https://vercel.com/dashboard)
3. Click "New Project"
4. Import your repository
5. Configure the project:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: None
   - Output Directory: ./
   - Install Command: pip install -r requirements.txt

6. Add the following environment variables:
   - `MONGODB_URI` - Your MongoDB Atlas connection string
   - `MONGODB_DB` - Your database name (e.g., `fyp_db`)
   - `MONGODB_COLLECTION` - Your collection name (e.g., `chat_history`)
   - `PINECONE_API_KEY` - Your Pinecone API key
   - `PINECONE_ENVIRONMENT` - Your Pinecone environment
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `ENVIRONMENT` - Set to `production`

7. Click "Deploy"

## Step 3: Connecting Your Frontend

After deployment, Vercel will provide you with a URL for your API. Update your frontend code to use this URL instead of localhost:

```python
# Change this in your code
FASTAPI_URL = "https://your-vercel-app-name.vercel.app/api"
```

## Step 4: Testing Your Deployment

1. Visit your Vercel deployment URL, which should show the API health check page
2. Test the API endpoints:
   - `https://your-vercel-app-name.vercel.app/api/docs` - API documentation
   - `https://your-vercel-app-name.vercel.app/api/upload_csv` - For data upload
   - `https://your-vercel-app-name.vercel.app/api/ask` - For chatbot queries

## Troubleshooting

1. **Logs**: Check the Vercel deployment logs for any errors
2. **Environment Variables**: Ensure all required environment variables are set correctly
3. **Dependencies**: Make sure all required dependencies are in `requirements.txt`
4. **Function Timeout**: Vercel has a 10-second timeout for serverless functions; optimize your code if needed
5. **MongoDB Connection**: Verify your MongoDB Atlas connection string and ensure network access is properly configured

## Limitations on Vercel's Free Tier

1. **Serverless Functions**: Limited to 10-second execution time
2. **Cold Starts**: Functions may experience cold starts
3. **Storage**: No persistent file system, use MongoDB for storage
4. **Memory**: Limited to 1GB RAM

## Next Steps

- Set up CORS properly for production
- Implement proper authentication and authorization
- Consider using a CDN for static assets
- Monitor your API performance and usage 