#!/bin/bash
# latexml_render 的容器中段：对 workspace/_latexml_render/in/*.xml 批量 latexmlpost。
# 图片（example.png 等）以相对路径留在原 tex 目录，须用 --sourcedirectory 指回
# workspace/<arxiv_id>/<时间戳>/，并加 --graphicimages 让其拷贝/转换进输出目录。
# 用法: docker compose run --rm --no-deps app bash xinda/cli/latexml_render_post.sh
set -u
IN=/app/workspace/_latexml_render/in
OUT=/app/workspace/_latexml_render/html
mkdir -p "$OUT"
total=$(ls "$IN"/*.xml | wc -l); i=0; fail=0
for f in "$IN"/*.xml; do
  name=$(basename "$f" .xml); i=$((i+1))
  pid=${name%%__*}
  srcdir=$(ls -d /app/workspace/"$pid"/*/ 2>/dev/null | tail -1)
  dest="$OUT/$name/$name.html"
  [ -s "$dest" ] && continue   # 断点续跑
  if ! latexmlpost --format=html5 --graphicimages ${srcdir:+--sourcedirectory="$srcdir"} \
       --destination="$dest" "$f" >/dev/null 2>&1; then
    fail=$((fail+1)); echo "FAIL $name"
  fi
  [ $((i % 10)) -eq 0 ] && echo "[$i/$total] done (fail=$fail)"
done
echo "latexmlpost finished: $i processed, $fail failed"
