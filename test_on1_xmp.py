#!/usr/bin/env python3
"""
Quick test to check ON1 XMP compatibility
"""
from pathlib import Path

def check_xmp_files():
    """Check XMP files in test directory"""
    # This path is for testing purposes only - users should modify this to their test folder
    test_dir = Path("./test")  # Default location for testing
    
    
    xmp_files = list(test_dir.glob("*.xmp"))
    
    print(f"🔍 Found {len(xmp_files)} XMP files")
    
    for xmp_file in xmp_files[:3]:  # Check first 3
        print(f"\n📄 {xmp_file.name}")
        
        try:
            with open(xmp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for ON1-compatible elements
            checks = {
                'XMP toolkit': 'x:xmptk=' in content,
                'Proper namespaces': 'xmlns:lr=' in content and 'xmlns:aux=' in content,
                'ModifyDate': 'xmp:ModifyDate=' in content,
                'CreatorTool': 'xmp:CreatorTool=' in content,
                'Keywords in bag': '<rdf:Bag>' in content and '<rdf:li>' in content,
                'Description in Alt': '<rdf:Alt>' in content and 'xml:lang=' in content,
                'AI keywords': 'AI:' in content,
                'Culler data': 'PhotoCuller:' in content
            }
            
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check}")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n💡 Next steps for ON1:")
    print(f"1. Restart ON1 Photo RAW completely")
    print(f"2. Go to File > Synchronize Folder")  
    print(f"3. Select your test folder")
    print(f"4. Check keywords panel for 'AI:' and 'PhotoCuller:' keywords")
    print(f"5. Check Description field for analysis")

if __name__ == "__main__":
    check_xmp_files()
