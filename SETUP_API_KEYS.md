# API Key Setup for Beta v3

## Quick Setup (Recommended)

### Option 1: Using .env file

1. **Copy the example file**:
   ```bash
   copy .env.beta_v3 .env
   ```

2. **Edit `.env` file** and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Install python-dotenv** (if not already installed):
   ```bash
   pip install python-dotenv
   ```

4. **Run Beta v3**:
   ```bash
   python run_beta_v3.py
   ```

### Option 2: Using Windows Environment Variable (Permanent)

1. **Open Environment Variables**:
   - Press `Win + R`
   - Type `sysdm.cpl` and press Enter
   - Click "Environment Variables" button

2. **Add new User variable**:
   - Click "New" under "User variables"
   - Variable name: `OPENAI_API_KEY`
   - Variable value: `sk-your-actual-api-key-here`
   - Click OK

3. **Restart your terminal** and run:
   ```bash
   python run_beta_v3.py
   ```

### Option 3: Using Command Line (Current Session Only)

**PowerShell**:
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
python run_beta_v3.py
```

**Command Prompt**:
```cmd
set OPENAI_API_KEY=sk-your-actual-api-key-here
python run_beta_v3.py
```

### Option 4: Using Setup Script

1. **Run the setup script**:
   ```cmd
   setup_api_key.bat
   ```

2. **Enter your API key** when prompted

3. **Run Beta v3** in the same terminal window

---

## Getting an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in or create an account
3. Go to **API Keys** section
4. Click **"Create new secret key"**
5. Copy the key (starts with `sk-`)
6. **Important**: Save it somewhere safe - you won't be able to see it again!

---

## Testing Your Setup

### Test with Mock Mode (No Skyrim Required)

```bash
# Test for 30 seconds with GPT-5
python run_beta_v3.py --test-mode --duration 30

# Verify API key is working
python -c "import os; print('✓ API key found' if os.getenv('OPENAI_API_KEY') else '✗ API key not found')"
```

### Expected Output

If API key is set correctly:
```
✓ Loaded environment from d:\Projects\Singularis\.env
...
✓ GPT-5 orchestrator initialized
```

If API key is missing:
```
⚠ GPT-5 enabled but no API key found
   Set OPENAI_API_KEY environment variable or run with --no-gpt5
   Continuing without GPT-5 coordination...
```

---

## Running Without GPT-5

If you don't have an API key or want to test without it:

```bash
python run_beta_v3.py --test-mode --no-gpt5 --duration 30
```

---

## Troubleshooting

### "API key not found" error

**Check if key is set**:
```powershell
# PowerShell
echo $env:OPENAI_API_KEY

# Command Prompt
echo %OPENAI_API_KEY%
```

**If empty**, the key is not set. Use one of the setup methods above.

### "Invalid API key" error

- Make sure your key starts with `sk-`
- Check for extra spaces or quotes
- Verify the key is active at [platform.openai.com](https://platform.openai.com)

### ".env file not loading"

**Install python-dotenv**:
```bash
pip install python-dotenv
```

**Verify .env file exists**:
```bash
dir .env
```

**Check .env file format** (no quotes needed):
```
OPENAI_API_KEY=sk-your-key-here
```

### "GPT-5 model not found" error

GPT-5 is currently in preview. If you don't have access:

1. **Use GPT-4 instead** - edit `run_beta_v3.py`:
   ```python
   gpt5_model: str = "gpt-4-turbo-preview"  # Change from "gpt-5"
   ```

2. **Or run without GPT-5**:
   ```bash
   python run_beta_v3.py --no-gpt5
   ```

---

## Security Best Practices

1. **Never commit .env to git**:
   - `.env` is already in `.gitignore`
   - Use `.env.beta_v3` as template only

2. **Keep API keys secret**:
   - Don't share in screenshots
   - Don't paste in public channels
   - Rotate keys if exposed

3. **Monitor usage**:
   - Check [platform.openai.com/usage](https://platform.openai.com/usage)
   - Set spending limits
   - Review API calls regularly

---

## Cost Estimates

**GPT-5 Pricing** (as of Nov 2025):
- Input: ~$0.01 per 1K tokens
- Output: ~$0.03 per 1K tokens

**Typical Beta v3 Usage**:
- ~500 tokens per coordination request
- ~10-20 coordinations per minute (with GPT-5 enabled)
- **Estimated cost**: $0.50-$1.00 per hour

**To reduce costs**:
- Use `--no-gpt5` for testing
- Increase `stats_interval` to reduce coordination frequency
- Use test mode instead of production

---

## Next Steps

Once your API key is set up:

1. **Run quick test**:
   ```bash
   python run_beta_v3.py --test-mode --duration 30
   ```

2. **Run full test suite**:
   ```bash
   python run_beta_v3_tests.py
   ```

3. **Run with Skyrim** (production mode):
   ```bash
   python run_beta_v3.py
   ```

---

**Need Help?**
- Check logs in `logs/` directory
- Review `BETA_V3_TESTING_GUIDE.md`
- See `BETA_V3_README.md` for full documentation
