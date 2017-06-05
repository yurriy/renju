all:
	sudo pip3 install keras h5py websockets
	wget -Osrc/trained_model https://www.dropbox.com/s/uzskj4jzbjjc1x6/trained_model
	git clone https://github.com/yyjhao/HTML5-Gomoku
	wget -OHTML5-Gomoku/js/Player.js https://www.dropbox.com/s/w2440y8rnzvml70/Player.js
