---
layout: post
title:      "Dealing with Outliers"
date:       2020-08-29 23:46:30 +0000
permalink:  dealing_with_outliers
---


This post will cover the topic of dealing with outliers in general, but focusing mostly on outliers in the context of multiple linear regression. I will outline and discuss several important questions that need to be asked with regards to outliers during the data scrubbing, exploring, and modeling phases, including:

## How can outliers be identified? What methods are available?
This section will provide a brief tutorial of useful methods for identifying possible outliers visually, as well as mathematically (z-score, IQR, Cook's distance).

## When should outliers be removed or retained?
This will vary based on the question(s) you are trying to address, the type of model you are using, whether the outliers seem to represent possible (albeit extreme) values or mistakes in the data collecting/recording process, etc.

## How does outlier retention/removal influence the quality and scope of my model?
