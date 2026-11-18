"""PyInstaller 用ランチャー（パッケージ内の相対importを使うため絶対importで起動する）。"""
from mk8dx.__main__ import main

if __name__ == "__main__":
    main()
