## 準備
リポジトリのルートに`presets.yaml`を設置して，以下のフォーマットに従って記述しておく．
```
presets:
  preset1:
    model: 'sonoisa/sentence-bert-base-ja-mean-tokens-v2' # 省略可能
    ref: "resources/preset1/ref.txt"
    base: "resources/preset1/base.txt"
    targets:
      method1: "resources/preset1/method1.txt"
      method2: "resources/preset1/method2.txt"
      method3: "resources/preset1/method3.txt"
      # 複数指定可能 

  preset2:
    ref: "resources/preset2/ref.txt"
    base: "resources/preset2/base.txt"
    targets:
      method1: "resources/preset2/method1.txt"
```

## 使用法
uvを使用して以下のように実行する．
```
uv run main.py --preset <preset-name>
```

