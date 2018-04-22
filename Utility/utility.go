package Utility

import (
	"encoding/gob"
	"io"
	"os"
)

func WriteGobToFile(filePath string, object interface{}) error {
	file, err := os.Create(filePath)
	if err == nil {
		WriteGob(file, object)
	}
	file.Close()
	return err
}

func WriteGob(w io.Writer, object interface{}) error {
	encoder := gob.NewEncoder(w)
	err := encoder.Encode(object)
	return err
}

func ReadGobFromFile(filePath string, object interface{}) error {
	file, err := os.Open(filePath)
	if err == nil {
		err = ReadGob(file, object)
	}
	file.Close()
	return err
}

func ReadGob(r io.Reader, object interface{}) error {
	decoder := gob.NewDecoder(r)
	err := decoder.Decode(object)
	return err
}
