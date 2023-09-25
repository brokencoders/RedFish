#pragma once 

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <unordered_map>

namespace Parser {

    bool isNumber(const std::string& str) {
        for (char c : str) {
            if (!isdigit(c) && c != '.' && c != '\r') {
                return false;
            }
        }
        return true;
    }

    void removeCharacters(std::string& str, char c)
    {
        size_t found = str.find(c);
        while (found != std::string::npos) {
            str.erase(found, 1);
            found = str.find(c, found);
        }
    }

    namespace CSV {
        // Basicaly every Row
        struct Element {
            std::vector<double> numbers;
            std::vector<std::string> strings;
        };

        struct File {
            // Name, Data Type (d = double, s = stirng)
            std::vector<std::pair<std::string, std::string>> headers;
            std::vector<Element> data;

            std::string path = nullptr;
            File();
            File(std::string path);
            void load(std::string path);
        };

        File::File() { }

        File::File(std::string path)
            :path(path)
        {
            load(path);
        }


        void File::load(std::string path)
        {
            std::ifstream file(path);

            if (!file.is_open())
                throw std::runtime_error("Error opening file: " + path + "\n");


                std::string line;
                
                // Get Data Names
                std::getline(file, line);
                std::istringstream iss(line);

                std::string token;
                while (std::getline(iss, token, ',')) {
                    removeCharacters(token, '\r');
                    headers.push_back({token, ""});
                }

                // Get Data Types and store first line
                std::getline(file, line);
                iss = std::istringstream(line);

                int i = 0;
                while (std::getline(iss, token, ',')) {
                    if (isNumber(token))
                        headers.at(i).second = "d";
                    else 
                        headers.at(i).second = "s";
                    i++;
                }
                while (std::getline(file, line)) {
                    iss = std::istringstream(line);

                    std::string token;
                    int i = 0;
                    Element el;
                    while (std::getline(iss, token, ',')) {
                        if(headers.at(i).second == "s")
                            el.strings.push_back(token);
                        else 
                            el.numbers.push_back(std::stod(token));
                        i++;
                    }
                    data.push_back(el);
                }
            file.close();
        }

        std::ostream& operator<<(std::ostream& os, const Element& e)
        {
            for(const auto& n : e.numbers)
                os << n << " ";
            for(const auto& s : e.strings)
                os << s << " "; 
            return os << std::endl;
        }


        std::ostream& operator<<(std::ostream& os, const File& f)
        {
            std::cout << "CSV File from: " << f.path << "\n";
            for(auto& heder : f.headers)
                os << heder.first << "(" << heder.second << ")\n";
            for (auto& element : f.data)
                os << element;
            return os;
        }
    }
}