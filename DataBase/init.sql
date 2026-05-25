CREATE DATABASE IF NOT EXISTS db;
USE db;

-- 사용자 정보 테이블
CREATE TABLE IF NOT EXISTS USER_TABLE (
    id BIGINT NOT NULL PRIMARY KEY,
    user_name varchar(255) NOT NULL,
    user_ref_audio_path varchar(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS CHAT_TABLE (
    massage_id varchar(255) NOT NULL,
    send_user_id BIGINT NOT NULL,
    rec_user_id BIGINT NOT NULL,
    massage varchar(255) NOT NULL,
    emotion_path varchar(255) NOT NULL,
    emotion INT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (send_user_id) REFERENCES USER_TABLE(id) ON DELETE CASCADE,
    FOREIGN KEY (rec_user_id) REFERENCES USER_TABLE(id) ON DELETE CASCADE,
    PRIMARY KEY (massage_id)
);