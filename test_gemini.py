#!/usr/bin/env python3
"""
Quick test script to verify Gemini API setup and test extraction.
"""

import os
import sys
from pathlib import Path

def test_dotenv():
    """Test if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv library imported successfully")
        load_dotenv()
        return True
    except ImportError as e:
        print(f"❌ Failed to import python-dotenv: {e}")
        print("\nRun: uv sync")
        return False

def test_env_file():
    """Test if .env file exists."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("\nCreate .env file:")
        print("  cp .env.example .env")
        print("  # Then edit .env and add your GEMINI_API_KEY")
        return False
    else:
        print("✓ .env file exists")
        return True

def test_api_key():
    """Test if API key is set."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("❌ GEMINI_API_KEY not set in .env file")
        print("\nTo get your API key:")
        print("1. Go to https://ai.google.dev/gemini-api/docs/api-key")
        print("2. Click 'Get API key'")
        print("3. Add it to .env file: GEMINI_API_KEY=your-key-here")
        return False
    else:
        print(f"✓ GEMINI_API_KEY is set (length: {len(api_key)})")
        return True

def test_import():
    """Test if google-genai is installed."""
    try:
        from google import genai
        print("✓ google-genai library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import google-genai: {e}")
        print("\nRun: uv sync")
        return False

def test_client():
    """Test if we can create a Gemini client."""
    try:
        from dotenv import load_dotenv
        from google import genai
        
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key or api_key == "your-api-key-here":
            return False
            
        client = genai.Client(api_key=api_key)
        print("✓ Gemini client created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False

def main():
    print("Testing Gemini API setup...")
    print("-" * 60)
    
    checks = [
        test_dotenv(),
        test_import(),
        test_env_file(),
        test_api_key(),
        test_client(),
    ]
    
    print("-" * 60)
    if all(checks):
        print("\n✓ All checks passed! Ready to use gemini_extract.py")
        print("\nTo test on 3 documents with 2 parallel workers:")
        print("  uv run python gemini_extract.py ce-fortaleza-2024.json --max-documents 3 --workers 2 --debug")
        print("\nTo process all documents with 5 parallel workers:")
        print("  uv run python gemini_extract.py ce-fortaleza-2024.json --workers 5")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
