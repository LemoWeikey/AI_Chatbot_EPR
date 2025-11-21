# Qdrant Cloud Setup Guide for EPR Chatbot

## Step 1: Create Qdrant Cloud Account

1. Go to https://cloud.qdrant.io/
2. Click "Sign Up" or "Get Started"
3. Create an account using your email or GitHub

## Step 2: Create a Cluster

1. After logging in, click **"Create Cluster"**
2. Choose your cluster configuration:
   - **Region**: Select closest to your users (e.g., AWS us-east-1, GCP europe-west1)
   - **Cluster Size**: Start with **Free tier (1GB)** for testing
   - For production, choose **1 node** or more based on your needs
3. Click **"Create"**
4. Wait for cluster to be provisioned (usually takes 1-2 minutes)

## Step 3: Get Your Credentials

After cluster creation, you'll see:

1. **Cluster URL**: Something like `https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east-1-0.aws.cloud.qdrant.io:6333`
2. **API Key**: Click **"Get API Key"** or **"Show Credentials"**
   - Copy this key - you won't see it again!
   - Store it securely

## Step 4: Configure Your Environment

1. Open your `.env` file in the project root
2. Add these lines:

```bash
# Qdrant Cloud Configuration
QDRANT_CLOUD_URL=https://your-cluster-url.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key-here
```

Example:
```bash
QDRANT_CLOUD_URL=https://abc12345-1234-5678-9abc-def123456789.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Step 5: Install Required Packages

Make sure you have the required packages:

```bash
pip install qdrant-client langchain-qdrant python-dotenv
```

## Step 6: Upload Your Data

We'll upload data in two stages: FAQ first, then Law.

### Upload FAQ Data

Run this command:
```bash
python upload_faq_to_qdrant.py
```

This will:
- Connect to your Qdrant Cloud cluster
- Create a collection called `faq_collection`
- Upload all FAQ Q&A pairs from `faq (1).json`
- Show progress and confirm upload

### Upload Law Data

After FAQ is successfully uploaded, run:
```bash
python upload_law_to_qdrant.py
```

This will:
- Create a collection called `law_collection`
- Upload all legal articles from `law.json`
- Show progress and confirm upload

## Step 7: Verify Upload

1. Go to Qdrant Cloud Dashboard
2. Click on your cluster
3. Go to **"Collections"** tab
4. You should see:
   - `faq_collection` with X points (number of FAQ items)
   - `law_collection` with Y points (number of law articles)

## Step 8: Test Your Chatbot

Run your chatbot:
```bash
streamlit run app.py
```

Test queries to ensure data is retrievable:
- FAQ test: "Các đối tượng nào phải thực hiện trách nhiệm tái chế?"
- Law test: "Điều 1 quy định về phạm vi điều chỉnh"

## Troubleshooting

### Connection Error
- Check your `QDRANT_CLOUD_URL` is correct (should start with `https://`)
- Verify API key is properly copied
- Ensure your cluster is **Running** (check dashboard)

### Authentication Error
- Your API key might be incorrect
- Regenerate API key from Qdrant Cloud dashboard
- Update `.env` file with new key

### Collection Already Exists
- The upload scripts will handle this automatically
- They check if collection exists before creating
- To force recreate, delete collections from dashboard first

### Slow Upload
- Large datasets take time (law.json is 770KB)
- Upload happens in batches for efficiency
- Check your internet connection
- Free tier has rate limits - consider upgrading if needed

## Cost Considerations

**Free Tier**:
- 1GB storage
- Good for ~100K-200K FAQ/Law entries
- Perfect for development and small production

**Paid Tier** (if you need more):
- Starts at ~$25/month for 1 node, 4GB RAM
- Scales based on data size and query volume
- See pricing at https://qdrant.io/pricing/

## Security Best Practices

1. Never commit `.env` file to git
2. Use environment variables in production
3. Rotate API keys periodically
4. Enable IP whitelisting if available
5. Use HTTPS only (default for Qdrant Cloud)

## Next Steps

After successful upload:
1. Test both FAQ and Law retrieval
2. Monitor query performance in Qdrant dashboard
3. Adjust retrieval parameters if needed (`top_k`, `score_threshold`)
4. Consider setting up backups for your data
5. Monitor usage and costs in Qdrant Cloud dashboard

## Support

- Qdrant Documentation: https://qdrant.tech/documentation/
- Qdrant Community: https://discord.gg/qdrant
- Issues: Check GitHub repository issues section
