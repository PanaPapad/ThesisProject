-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Apr 25, 2023 at 02:58 PM
-- Server version: 10.4.27-MariaDB
-- PHP Version: 8.0.25

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

--
-- Database: `ids_storage`
--
CREATE DATABASE IF NOT EXISTS `ids_storage` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `ids_storage`;

-- --------------------------------------------------------

--
-- Table structure for table `processed_data`
--

CREATE TABLE `processed_data` (
  `id` bigint(20) NOT NULL,
  `raw_data_id` bigint(20) NOT NULL,
  `insert_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `data` mediumblob DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `raw_data`
--

CREATE TABLE `raw_data` (
  `id` bigint(20) NOT NULL,
  `insert_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `data` mediumblob DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `results`
--

CREATE TABLE `results` (
  `id` bigint(20) NOT NULL,
  `processed_data_id` bigint(20) NOT NULL,
  `insert_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `data` mediumblob DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `processed_data`
--
ALTER TABLE `processed_data`
  ADD PRIMARY KEY (`id`),
  ADD KEY `raw_data_id` (`raw_data_id`);

--
-- Indexes for table `raw_data`
--
ALTER TABLE `raw_data`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `results`
--
ALTER TABLE `results`
  ADD PRIMARY KEY (`id`),
  ADD KEY `processed_data_id` (`processed_data_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `processed_data`
--
ALTER TABLE `processed_data`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `raw_data`
--
ALTER TABLE `raw_data`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `results`
--
ALTER TABLE `results`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `processed_data`
--
ALTER TABLE `processed_data`
  ADD CONSTRAINT `processed_data_ibfk_1` FOREIGN KEY (`raw_data_id`) REFERENCES `raw_data` (`id`);

--
-- Constraints for table `results`
--
ALTER TABLE `results`
  ADD CONSTRAINT `results_ibfk_1` FOREIGN KEY (`processed_data_id`) REFERENCES `processed_data` (`id`);
COMMIT;
