drop database if exists dia;
create database dia;
use dia;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    name VARCHAR(225),
    email VARCHAR(50), 
    password VARCHAR(50)
    );
