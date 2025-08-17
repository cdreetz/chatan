#!/usr/bin/env python3
"""Launch the Chatan React viewer for building datasets through the UI."""

import asyncio
import sys
from chatan.viewer.live_viewer_new import LiveViewer


async def main():
    """Launch the viewer with optional LLM configuration."""
    
    print("🚀 Starting Chatan Dataset Builder")
    print("=" * 50)
    print()
    
    # Ask user for LLM configuration
    print("Configure LLM Provider (optional - you can also use mock data):")
    print("1. OpenAI")
    print("2. Anthropic") 
    print("3. Skip (use mock data for testing)")
    print()
    
    choice = input("Choose option (1-3) [3]: ").strip() or "3"
    
    provider = None
    api_key = None
    
    if choice == "1":
        provider = "openai"
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("OPENAI_API_KEY not found. Enter OpenAI API key: ").strip()
            if not api_key:
                print("No API key provided, using mock data")
                provider = None
        else:
            print(f"✅ Found OPENAI_API_KEY environment variable")
    elif choice == "2":
        provider = "anthropic"
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = input("ANTHROPIC_API_KEY not found. Enter Anthropic API key: ").strip()
            if not api_key:
                print("No API key provided, using mock data")
                provider = None
        else:
            print(f"✅ Found ANTHROPIC_API_KEY environment variable")
    
    print()
    print("🎯 What you can do:")
    print("• Start with a blank spreadsheet")
    print("• Click 'Add Your First Column' to define data types")
    print("• Choose from AI Prompts, Choice Lists, or References")
    print("• Generate realistic data and export it")
    print("• Everything works through the visual interface!")
    print()
    
    if provider:
        print(f"✅ Using {provider.upper()} for AI generation")
    else:
        print("🧪 Using mock data - great for testing!")
    
    print()
    print("Starting viewer... Press Ctrl+C to stop")
    print()
    
    try:
        # Launch the viewer
        viewer = LiveViewer(title="Chatan Dataset Builder", auto_open=True)

        if provider and api_key:
            viewer.set_generator_client(provider, api_key)

        url = await viewer.start({})

        
        print("✅ Viewer is running!")
        print("📱 The interface should have opened in your browser")
        print(f"🌐 URL: {url}")
        print()
        print("💡 Tips:")
        print("• Click column headers to edit them")
        print("• Use {column_name} syntax to reference other columns")
        print("• Try different column types to see what works best")
        print()
        print("Server running... Press Ctrl+C to stop")
        
        # Keep running until user stops
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        if 'viewer' in locals():
            await viewer.stop()
        print("✅ Stopped successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def validate_environment():
    """Check if the environment is set up correctly."""
    try:
        import chatan
        from chatan.viewer.live_viewer_new import LiveViewer
        print("✅ Chatan package found")
        return True
    except ImportError as e:
        print(f"❌ Chatan package not found: {e}")
        print("Please install chatan: pip install -e .")
        return False


if __name__ == "__main__":
    print("🧪 Chatan React Viewer")
    print("=" * 30)
    
    if not validate_environment():
        sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)
