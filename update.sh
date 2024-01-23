echo "clearing the dist directory ... "
rm -f dist/*

echo "generating the whl file ... "
python setup.py sdist bdist_wheel

echo "uninstalling old version ... "
pip uninstall -y kis

echo "installing new version ... "
pip install dist/*.whl

echo "clear the temp files ..."


echo "showing the information of kis ... "
pip show kis
