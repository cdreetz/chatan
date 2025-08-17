#!/usr/bin/env python3
"""Launch the React viewer with a blank sheet for building datasets through the UI."""

import asyncio
from chatan.viewer.live_viewer import LiveViewer


async def launch_blank_viewer():
    """Start the React viewer with an empty schema."""
    
    print("🚀 Starting Chatan Dataset Builder")
    print("=" * 40)
    print("Opening blank spreadsheet interface...")
    print("You can:")
    print("• Click column headers to define generation logic")
    print("• Add new columns with the + button")
    print("• Use the Generate button to create data")
    print("• Export your dataset when ready")
    print("\nPress Ctrl+C to stop the server")
    print()
    
    # Create viewer with empty schema
    viewer = LiveViewer(
        title="Dataset Builder - New Dataset", 
        auto_open=True
    )
    
    try:
        # Start with empty schema - user will build it through UI
        empty_schema = {}
        
        url = await viewer.start(empty_schema)
        print(f"✅ Viewer started at: {url}")
        print("The interface should have opened in your browser")
        print("\nServer running... Press Ctrl+C to stop")
        
        # Keep the server running until user stops it
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down viewer...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await viewer.stop()
        print("✅ Viewer stopped")


if __name__ == "__main__":
    asyncio.run(launch_blank_viewer())
