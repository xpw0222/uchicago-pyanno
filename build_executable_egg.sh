rm -r build
rm -r dist
python setup.py bdist_egg
chmod a+x dist/*egg
cp -r pyanno-gui.app dist/
cp dist/*egg dist/pyanno-gui.app/Contents/Resources
cp -r data dist
